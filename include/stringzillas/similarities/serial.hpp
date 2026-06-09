/**
 *  @brief ISA-agnostic parallel string-similarity template core (serial backend).
 *  @file include/stringzillas/similarities/serial.hpp
 *  @author Ash Vardanian
 *
 *  Contains the shared template core for all parallel similarity backends:
 *  Algorithm Building Blocks, the base `tile_scorer`, `diagonal_walker`/`horizontal_walker`,
 *  cost types, and substitution matrices, plus the serial backend aliases.
 *
 *  Every backend specialization header (`icelake.hpp`, `cuda.cuh`, ...) must include this
 *  file first, so that the primary templates are visible before any specialization.
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
constexpr size_t error_cost_abs(error_cost_t x) noexcept { return static_cast<size_t>(x < 0 ? -(i32_t)x : (i32_t)x); }

/**
 *  @brief A trivial function object for linear and affine gap costs in Levenshtein-like similarity algorithms.
 *  @sa affine_gap_costs_t
 */
struct linear_gap_costs_t {
    error_cost_t open_or_extend = 1;

    constexpr size_t magnitude() const noexcept { return error_cost_abs(open_or_extend); }
};

/**
 *  @brief A trivial function object for affine gap costs in Levenshtein-like similarity algorithms.
 *  @sa linear_gap_costs_t
 */
struct affine_gap_costs_t {
    error_cost_t open = 1;
    error_cost_t extend = 1;

    constexpr size_t magnitude() const noexcept { return std::max(error_cost_abs(open), error_cost_abs(extend)); }
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
    constexpr error_cost_t operator()(sz_rune_t a, sz_rune_t b) const noexcept { return a == b ? match : mismatch; }
    constexpr size_t magnitude() const noexcept { return std::max(error_cost_abs(match), error_cost_abs(mismatch)); }
};

/**
 *  @brief The number of distinct character classes in a @b `error_costs_32x32_t` table.
 *         Fixed at 32 to cover the largest practical alphabets - ~4 DNA bases, ~20 amino-acids,
 *         or keyboard-distance buckets - while keeping the table tiny.
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

    sz_u8_t byte_to_class[256] = {0};
    error_cost_t class_substitution_costs[classes_count_k][classes_count_k] = {{0}};

    constexpr error_cost_t operator()(char a, char b) const noexcept {
return class_substitution_costs[byte_to_class[(sz_u8_t)a]][byte_to_class[(sz_u8_t)b]];
}
    constexpr error_cost_t operator()(sz_u8_t a, sz_u8_t b) const noexcept {
        return class_substitution_costs[byte_to_class[a]][byte_to_class[b]];
    }

    constexpr size_t magnitude() const noexcept {
        size_t max_magnitude = 0;
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
     *         assigning one class per @b used residue (those with a non-placeholder diagonal cost).
     *
     *  Class 0 is the catch-all bucket for every byte that is not a used uppercase residue and costs
     *  zero against everything. Each used residue receives a distinct class index, so BLOSUM62 (≤20
     *  amino-acids) and NUC.4.4 (≤5 nucleotides) both fit comfortably within the 32 available classes.
     */
    static constexpr error_costs_32x32_t from_ascii_26x26_(error_cost_t const (&cells)[26][26]) noexcept {
        constexpr error_cost_t na = -128; // Placeholder for unused residues
        error_costs_32x32_t result;

        // Assign a fresh class to every residue whose diagonal cost is not a placeholder.
        sz_u8_t residue_to_class[26] = {0};
        sz_u8_t next_class = 1;
        for (int i = 0; i != 26; ++i)
            if (cells[i][i] != na) {
                residue_to_class[i] = next_class;
                result.byte_to_class[i + 65] = next_class;
                ++next_class;
            }

        // Populate the cost between every pair of used residues.
        for (int i = 0; i != 26; ++i)
            for (int j = 0; j != 26; ++j)
                if (residue_to_class[i] != 0 && residue_to_class[j] != 0)
                    result.class_substitution_costs[residue_to_class[i]][residue_to_class[j]] = cells[i][j];
        return result;
    }
};

#pragma region - Algorithm Building Blocks

/**
 *  @brief Helper object to guess the amount of SRAM we want to effectively process the input
 *         without fetching from RAM/VRAM all the time, including the space for 3 diagonals
 *         and the strings themselves.
 *
 *  @tparam size_type_ The type of the size, usually `size_t` for large inputs or `unsigned` on small inputs in CUDA.
 *  @tparam is_signed_ Whether the similarity scores can be negative or not.
 */
template <typename size_type_, bool is_signed_>
struct similarity_memory_requirements {
    using size_t = size_type_;
    static constexpr bool is_signed_k = is_signed_;

    size_t max_diagonal_length = 0;
    bytes_per_cell_t bytes_per_cell = zero_bytes_per_cell_k;
    size_t bytes_per_diagonal = 0;
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
    constexpr similarity_memory_requirements(              //
        size_t first_length, size_t second_length,         //
        sz_similarity_gaps_t gap_type,                     //
        size_t substitute_magnitude, size_t gap_magnitude, //
        size_t bytes_per_char,                             //
        size_t register_width,                             //
        bytes_per_cell_t min_bytes_per_cell = one_byte_per_cell_k) noexcept {

        // If any of the strings is empty, we don't need any memory to perform the similarity scoring.
        size_t shorter_length = sz_min_of_two(first_length, second_length);
        if (shorter_length == 0) {
            this->max_diagonal_length = 0;
            this->bytes_per_cell = zero_bytes_per_cell_k;
            this->bytes_per_diagonal = 0;
            this->total = 0;
            return;
        }

        // Each diagonal in the DP matrix is only by 1 longer than the shorter string.
        size_t longer_length = sz_max_of_two(first_length, second_length);
        this->max_diagonal_length = shorter_length + 1;

        // The amount of memory we need per diagonal, depends on the maximum number of the differences
        // between 2 strings and the maximum cost of each change.
        size_t magnitude = sz_max_of_two(substitute_magnitude, gap_magnitude);
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
        // When dealing with affine gaps, we need 3x diagonals of 1 matrix and 2x diagonals of 2 matrices.
        size_t diagonals_count = gap_type == sz_gaps_linear_k ? 3 : 7;
        size_t first_length_bytes = round_up_to_multiple<size_t>(first_length * bytes_per_char, register_width);
        size_t second_length_bytes = round_up_to_multiple<size_t>(second_length * bytes_per_char, register_width);
        this->total = diagonals_count * bytes_per_diagonal + first_length_bytes + second_length_bytes;
    }
};

#pragma region - Core Templates

#if SZ_HAS_CONCEPTS_

template <typename iterator_type_>
concept pointer_like = requires(iterator_type_ iterator, std::size_t idx) {
    { ++iterator } -> std::same_as<iterator_type_ &>; // pre-increment
    { *iterator };                                    // dereference
    { iterator[idx] };                                // random access
};

template <typename value_type_>
concept score_like = std::integral<value_type_> && std::is_trivial_v<value_type_>;

template <typename substituter_type_>
concept substituter_like = requires(substituter_type_ costs) {
    { costs.magnitude() } -> std::convertible_to<size_t>;                      // retrieving the magnitude
    { costs.operator()(char(), char()) } -> std::convertible_to<error_cost_t>; // cost of substitution
};

template <typename gap_costs_type_>
concept gap_costs_like = requires(gap_costs_type_ costs) {
    { costs.magnitude() } -> std::convertible_to<size_t>; // retrieving the magnitude
};

#endif

/**
 *  @brief An operator to be applied to be applied to all @b 2x2 tiles of the DP matrix to produce
 *         the bottom-right value from the 3x others when populating the Dynamic Programming matrix.
 *
 *  @tparam first_iterator_type_ Typically `char*`, `sz_rune_t*`, or a `constant_iterator`.
 *  @tparam second_iterator_type_ Typically `char*` or `sz_rune_t*`.
 *  @tparam score_type_ The type of the score, typically `size_t` or `sz_ssize_t`.
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
 *          @b (anti)diagonal-by-(anti)diagonal on a CPU.
 *
 *  Can be used for both global and local alignment, like Needleman-Wunsch and Smith-Waterman.
 *  Can be used for both linear and affine gap penalties.
 *
 *  ? There are smarter algorithms for computing the Levenshtein distance, mostly based on bit-level operations.
 *  ? Those, however, don't generalize well to arbitrary length inputs or non-uniform substitution costs.
 *  ? This algorithm provides a more flexible baseline implementation for future SIMD and GPGPU optimizations.
 *
 *  @tparam char_type_ The type of the characters in the strings, generally `char` or @b `rune_t` for UTF-8.
 *  @tparam score_type_ The smallest type that can hold the distance, ideally `sz_i8_t` or `sz_u8_t`.
 *  @tparam substituter_type_ A callable type that takes two characters and returns the substitution cost.
 *  @tparam gap_costs_type_ Whether to use linear or affine gap penalties.
 *  @tparam allocator_type_ A default-constructible allocator type for the internal buffers.
 *  @tparam objective_ Whether to minimize the distance or maximize the score.
 *  @tparam locality_ Whether to use the global alignment algorithm or the local one.
 *  @tparam capability_ Whether to use @b multi-threading or some form of @b SIMD vectorization, or both.
 *  @tparam enable_ Used to enable/disable the specialization.
 */
template <                                                       //
    typename char_type_ = char,                                  //
    typename score_type_ = size_t,                               //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    typename gap_costs_type_ = linear_gap_costs_t,               //
    typename allocator_type_ = dummy_alloc_t,                    //
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
 *          @b row-by-row on a CPU, using the conventional Wagner-Fischer algorithm.
 *
 *  Can be used for both global and local alignment, like Needleman-Wunsch and Smith-Waterman.
 *  Can be used for both linear and affine gap penalties.
 *
 *  @tparam char_type_ The type of the characters in the strings, generally `char` or @b `rune_t` for UTF-8.
 *  @tparam score_type_ The smallest type that can hold the distance, ideally `sz_i8_t` or `sz_u8_t`.
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
    typename char_type_ = char,                                  //
    typename score_type_ = size_t,                               //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    typename gap_costs_type_ = linear_gap_costs_t,               //
    typename allocator_type_ = dummy_alloc_t,                    //
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
 *  @brief Computes one or many pairwise Levenshtein distances in parallel using the CPU backend.
 *         For pairs of very large strings, all cores cooperate to compute one distance maximizing
 *         cache hits. For smaller strings, each core computes its own distance.
 */
template <                                         //
    typename char_type_ = char,                    //
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
    typename char_type_ = char,                    //
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
    typename char_type_ = char,                       //
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
    typename char_type_ = char,                       //
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
using levenshtein_serial_t = levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k>;
using levenshtein_utf8_serial_t = levenshtein_distances_utf8<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k>;
using needleman_wunsch_serial_t =
    needleman_wunsch_scores<char, error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k>;
using smith_waterman_serial_t =
    smith_waterman_scores<char, error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k>;

using affine_levenshtein_serial_t = levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_cap_serial_k>;
using affine_levenshtein_utf8_serial_t =
    levenshtein_distances_utf8<char, affine_gap_costs_t, malloc_t, sz_cap_serial_k>;
using affine_needleman_wunsch_serial_t =
    needleman_wunsch_scores<char, error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k>;
using affine_smith_waterman_serial_t =
    smith_waterman_scores<char, error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k>;

/**
 *  In @b AVX-512:
 *  - for Global Alignments, we can vectorize the min-max calculation for diagonal "walkers"
 *  - for Local Alignments, we can vectorize the character substitution lookups for horizontal "walkers"
 */
using levenshtein_icelake_t = levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_caps_sil_k>;
using levenshtein_utf8_icelake_t = levenshtein_distances_utf8<char, linear_gap_costs_t, malloc_t, sz_caps_sil_k>;
using needleman_wunsch_icelake_t =
    needleman_wunsch_scores<char, error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_caps_sil_k>;
using smith_waterman_icelake_t =
    smith_waterman_scores<char, error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_caps_sil_k>;

using affine_levenshtein_icelake_t = levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_caps_sil_k>;
using affine_levenshtein_utf8_icelake_t = levenshtein_distances_utf8<char, affine_gap_costs_t, malloc_t, sz_caps_sil_k>;

// TODO: Ice Lake optimizations don't yield massive improvements, but can be added later.
// using affine_needleman_wunsch_icelake_t =
//     needleman_wunsch_scores<char, error_costs_256x256_t, affine_gap_costs_t, malloc_t, sz_caps_sil_k>;
// using affine_smith_waterman_icelake_t =
//     smith_waterman_scores<char, error_costs_256x256_t, affine_gap_costs_t, malloc_t, sz_caps_sil_k>;

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

  public:
    tile_scorer() = default;
    tile_scorer(substituter_t subs, linear_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

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

        error_cost_t const gap_cost = gap_costs_.open_or_extend;

        executor.for_n(n, [&](size_t i) noexcept {
            score_t pre_substitution = scores_pre_substitution[i];
            score_t pre_insertion = scores_pre_insertion[i];
            score_t pre_deletion = scores_pre_deletion[i];

            // ? Note that here we are still traversing both buffers in the same order,
            // ? because one of the strings has been reversed beforehand.
            error_cost_t cost_of_substitution = substituter_(first_reversed_slice[i], second_slice[i]);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_deletion_or_insertion = min_or_max<objective_k>(pre_deletion, pre_insertion) + gap_cost;
            score_t cell_score = min_or_max<objective_k>(if_deletion_or_insertion, if_substitution);
            scores_new[i] = cell_score;
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

  public:
    tile_scorer() = default;
    tile_scorer(substituter_t subs, linear_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

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

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                           //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice, size_t const n, //
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion, score_t const *scores_pre_deletion,
        score_t *scores_new, executor_type_ &&executor = {}) noexcept {

        error_cost_t const gap_cost = gap_costs_.open_or_extend;
        std::atomic<score_t> atomic_best_score {best_score_};
        executor.for_slices(n, [&](size_t i_start, size_t i_end) noexcept {
            score_t local_best_score = atomic_best_score;
            for (size_t i = i_start; i < i_end; ++i) {
                score_t pre_substitution = scores_pre_substitution[i];
                score_t pre_insertion = scores_pre_insertion[i];
                score_t pre_deletion = scores_pre_deletion[i];

                // ? Note that here we are still traversing both buffers in the same order,
                // ? because one of the strings has been reversed beforehand.
                error_cost_t cost_of_substitution = substituter_(first_reversed_slice[i], second_slice[i]);
                score_t if_substitution = pre_substitution + cost_of_substitution;
                score_t if_deletion_or_insertion = min_or_max<objective_k>(pre_deletion, pre_insertion) + gap_cost;
                // ! This is the main difference with global alignment:
                score_t if_substitution_or_reset = min_or_max<objective_k, score_t>(if_substitution, 0);
                score_t cell_score = min_or_max<objective_k>(if_deletion_or_insertion, if_substitution_or_reset);
                scores_new[i] = cell_score;

                // ! Update the global maximum score if this cell beats it - this is the costliest operation:
                local_best_score = min_or_max<objective_k>(local_best_score, cell_score);
            }
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

  public:
    tile_scorer() = default;
    tile_scorer(substituter_t subs, affine_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

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

        executor.for_n(n, [&](size_t i) noexcept {
            score_t pre_substitution = scores_pre_substitution[i];
            score_t pre_insertion_opening = scores_pre_insertion[i];
            score_t pre_deletion_opening = scores_pre_deletion[i];
            score_t pre_insertion_expansion = scores_running_insertions[i];
            score_t pre_deletion_expansion = scores_running_deletions[i];

            // ? Note that here we are still traversing both buffers in the same order,
            // ? because one of the strings has been reversed beforehand.
            error_cost_t cost_of_substitution = substituter_(first_reversed_slice[i], second_slice[i]);
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

  public:
    tile_scorer() = default;
    tile_scorer(substituter_t subs, affine_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

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
            score_t local_best_score = atomic_best_score;
            for (size_t i = i_start; i < i_end; ++i) {
                score_t pre_substitution = scores_pre_substitution[i];
                score_t pre_insertion_opening = scores_pre_insertion[i];
                score_t pre_deletion_opening = scores_pre_deletion[i];
                score_t pre_insertion_expansion = scores_running_insertions[i];
                score_t pre_deletion_expansion = scores_running_deletions[i];

                // ? Note that here we are still traversing both buffers in the same order,
                // ? because one of the strings has been reversed beforehand.
                error_cost_t cost_of_substitution = substituter_(first_reversed_slice[i], second_slice[i]);
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
                local_best_score = min_or_max<objective_k>(local_best_score, cell_score);
            }
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
template <typename char_type_, typename score_type_, typename substituter_type_, typename allocator_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_, sz_capability_t capability_,
          typename enable_>
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && substituter_like<substituter_type_>
#endif
struct diagonal_walker<char_type_, score_type_, substituter_type_, linear_gap_costs_t, allocator_type_, objective_,
                       locality_, capability_, enable_> {

    using char_t = char_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = capability_;

    using allocated_t = typename allocator_t::value_type;
    static_assert(sizeof(allocated_t) == sizeof(char), "Allocator must be byte-aligned");
    using tile_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    mutable allocator_t alloc_ {};

    diagonal_walker(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}

    /**
     *  @param[in] subs A commutative function returning the cost of substituting one char with another.
     *  @param[in] gaps The uniform cost of a gap (insertion or deletion).
     *  @param[in] alloc A default-constructible allocator for the internal buffers.
     */
    diagonal_walker(substituter_t subs, linear_gap_costs_t gaps, allocator_t alloc) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

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
                        executor_type_ &&executor = {}) const noexcept {

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

        // We want to avoid reverse-order iteration over the shorter string.
        // Let's allocate a bit more memory and reverse-export our shorter string into that buffer.
        size_t const buffer_length = sizeof(score_t) * max_diagonal_length * 3 + shorter_length * sizeof(char_t);
        score_t *const buffer = (score_t *)alloc_.allocate(buffer_length);
        if (!buffer) return status_t::bad_alloc_k;

        // The next few pointers will be swapped around.
        score_t *previous_scores = buffer;
        score_t *current_scores = previous_scores + max_diagonal_length;
        score_t *next_scores = current_scores + max_diagonal_length;
        char_t *const shorter_reversed = (char_t *)(next_scores + max_diagonal_length);

        // Export the reversed string into the buffer.
        for (size_t i = 0; i != shorter_length; ++i) shorter_reversed[i] = shorter[shorter_length - 1 - i];

        // Initialize the first two diagonals:
        tile_scorer_t scorer {substituter_, gap_costs_};
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
            sz_move_serial((sz_ptr_t)(previous_scores), (sz_ptr_t)(previous_scores + 1),
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
        alloc_.deallocate((allocated_t *)buffer, buffer_length);
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
template <typename char_type_, typename score_type_, typename substituter_type_, typename allocator_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_, sz_capability_t capability_,
          typename enable_>
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && substituter_like<substituter_type_>
#endif
struct diagonal_walker<char_type_, score_type_, substituter_type_, affine_gap_costs_t, allocator_type_, objective_,
                       locality_, capability_, enable_> {

    using char_t = char_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = affine_gap_costs_t;
    using allocator_t = allocator_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_gaps_t gaps_k = sz_gaps_affine_k;
    static constexpr sz_capability_t capability_k = capability_;

    using allocated_t = typename allocator_t::value_type;
    static_assert(sizeof(allocated_t) == sizeof(char), "Allocator must be byte-aligned");
    using tile_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    mutable allocator_t alloc_ {};

    diagonal_walker(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}

    /**
     *  @param[in] subs A commutative function returning the cost of substituting one char with another.
     *  @param[in] gaps The affine costs of opening and extending a gap.
     *  @param[in] alloc A default-constructible allocator for the internal buffers.
     */
    diagonal_walker(substituter_t subs, affine_gap_costs_t gaps, allocator_t alloc) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

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
                        executor_type_ &&executor = {}) const noexcept {

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

        // We want to avoid reverse-order iteration over the shorter string.
        // Let's allocate a bit more memory and reverse-export our shorter string into that buffer.
        size_t const buffer_length = sizeof(score_t) * max_diagonal_length * 7 + shorter_length * sizeof(char_t);
        score_t *const buffer = (score_t *)alloc_.allocate(buffer_length);
        if (!buffer) return status_t::bad_alloc_k;

        // The next few pointers will be swapped around.
        score_t *previous_scores = buffer;
        score_t *current_scores = previous_scores + max_diagonal_length;
        score_t *next_scores = current_scores + max_diagonal_length;
        score_t *current_inserts = next_scores + max_diagonal_length;
        score_t *next_inserts = current_inserts + max_diagonal_length;
        score_t *current_deletes = next_inserts + max_diagonal_length;
        score_t *next_deletes = current_deletes + max_diagonal_length;
        char_t *const shorter_reversed = (char_t *)(next_deletes + max_diagonal_length);

        // Export the reversed string into the buffer.
        for (size_t i = 0; i != shorter_length; ++i) shorter_reversed[i] = shorter[shorter_length - 1 - i];

        // Initialize the first two diagonals:
        tile_scorer_t scorer {substituter_, gap_costs_};
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
            sz_move_serial((sz_ptr_t)(previous_scores), (sz_ptr_t)(previous_scores + 1),
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
        alloc_.deallocate((allocated_t *)buffer, buffer_length);
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
template <typename char_type_, typename score_type_, typename substituter_type_, typename allocator_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && substituter_like<substituter_type_>
#endif
struct horizontal_walker<char_type_, score_type_, substituter_type_, linear_gap_costs_t, allocator_type_, objective_,
                         locality_, sz_cap_serial_k, void> {

    using char_t = char_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;
    using walker_t = horizontal_walker<char_t, score_t, substituter_t, gap_costs_t, allocator_t, objective_k,
                                       locality_k, capability_k, void>;

    using allocated_t = typename allocator_t::value_type;
    static_assert(sizeof(allocated_t) == sizeof(char), "Allocator must be byte-aligned");
    using tile_scorer_t = tile_scorer<constant_iterator<char_t>, char_t const *, score_t, substituter_t, gap_costs_t,
                                      objective_k, locality_k, capability_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    mutable allocator_t alloc_ {};

    horizontal_walker(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}

    /**
     *  @param[in] subs A commutative function returning the cost of substituting one char with another.
     *  @param[in] gaps The uniform cost of a gap (insertion or deletion).
     *  @param[in] alloc A default-constructible allocator for the internal buffers.
     *
     */
    horizontal_walker(substituter_t subs, linear_gap_costs_t gaps, allocator_t alloc) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

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
                        executor_type_ &&executor = {}) const noexcept {

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

        // We decide to use less memory!
        size_t const buffer_length = sizeof(score_t) * shorter_dim * 2;
        score_t *const buffer = (score_t *)alloc_.allocate(buffer_length);
        if (!buffer) return status_t::bad_alloc_k;

        // The next few pointers will be swapped around.
        score_t *previous_scores = buffer;
        score_t *current_scores = previous_scores + shorter_dim;

        // Initialize the first row:
        tile_scorer_t scorer {substituter_, gap_costs_};
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
        alloc_.deallocate((allocated_t *)buffer, buffer_length);
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
template <typename char_type_, typename score_type_, typename substituter_type_, typename allocator_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && substituter_like<substituter_type_>
#endif
struct horizontal_walker<char_type_, score_type_, substituter_type_, affine_gap_costs_t, allocator_type_, objective_,
                         locality_, sz_cap_serial_k, void> {

    using char_t = char_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = affine_gap_costs_t;
    using allocator_t = allocator_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;
    using walker_t = horizontal_walker<char_t, score_t, substituter_t, gap_costs_t, allocator_t, objective_k,
                                       locality_k, capability_k, void>;

    using allocated_t = typename allocator_t::value_type;
    static_assert(sizeof(allocated_t) == sizeof(char), "Allocator must be byte-aligned");
    using tile_scorer_t = tile_scorer<constant_iterator<char_t>, char_t const *, score_t, substituter_t, gap_costs_t,
                                      objective_k, locality_k, capability_k>;

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    mutable allocator_t alloc_ {};

    horizontal_walker(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}

    /**
     *  @param[in] subs A commutative function returning the cost of substituting one char with another.
     *  @param[in] gap_opening_cost The cost of opening a gap (insertion or deletion).
     *  @param[in] gap_extension_cost The cost of extending a gap (insertion or deletion).
     *  @param[in] alloc A default-constructible allocator for the internal buffers.
     */
    horizontal_walker(substituter_t subs, affine_gap_costs_t gaps, allocator_t alloc) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

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
                        executor_type_ &&executor = {}) const noexcept {

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

        // We decide to use less memory!
        size_t const buffer_length = sizeof(score_t) * shorter_dim * 2 * 3; // 2x rows of 3x matrices
        score_t *const buffer = (score_t *)alloc_.allocate(buffer_length);
        if (!buffer) return status_t::bad_alloc_k;

        // The next few pointers will be swapped around.
        score_t *previous_scores = buffer;
        score_t *current_scores = previous_scores + shorter_dim;
        score_t *previous_inserts = current_scores + shorter_dim;
        score_t *current_inserts = previous_inserts + shorter_dim;
        score_t *previous_deletes = current_inserts + shorter_dim;
        score_t *current_deletes = previous_deletes + shorter_dim;

        // Initialize the first row:
        tile_scorer_t scorer {substituter_, gap_costs_};
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
        alloc_.deallocate((allocated_t *)buffer, buffer_length);
        return status_t::success_k;
    }
};

#pragma endregion - Horizontal Walker

#pragma endregion - Algorithm Building Blocks

#pragma region - Pairwise Algorithms on CPU

/**
 *  @brief Computes the @b byte-level Levenshtein distance between two strings using the CPU backend.
 *  @sa `levenshtein_distance_utf8` for UTF-8 strings.
 *
 *  @tparam char_type_ Can be any POD integer type, but @b `char` and @b `sz_rune_t` are preferred.
 *  @tparam gap_costs_type_ Can be either `linear_gap_costs_t` or `affine_gap_costs_t`.
 *  @tparam capability_ Can be either `sz_cap_serial_k`, `sz_caps_sil_k`, `sz_cap_cuda_k`.
 */
template <                                         //
    typename char_type_ = char,                    //
    typename gap_costs_type_ = linear_gap_costs_t, //
    typename allocator_type_ = dummy_alloc_t,      //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename enable_ = void                        //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct levenshtein_distance {

    using char_t = char_type_;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = serialize_capability(capability_k);

    using horizontal_u8_t =                                                                        //
        horizontal_walker<char_t, sz_u8_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                          sz_minimize_distance_k, sz_similarity_global_k, capability_serialized_k>;
    using diagonal_u8_t =                                                                        //
        diagonal_walker<char_t, sz_u8_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_serialized_k>;
    using diagonal_u16_t =                                                                        //
        diagonal_walker<char_t, sz_u16_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u32_t =                                                                        //
        diagonal_walker<char_t, sz_u32_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u64_t =                                                                        //
        diagonal_walker<char_t, sz_u64_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;

    using linearized_fallback_t = levenshtein_distance<char_t, linear_gap_costs_t, allocator_t, capability_k>;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    levenshtein_distance(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    levenshtein_distance(uniform_substitution_costs_t subs, gap_costs_t gaps,
                         allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, size_t &result_ref,
                        executor_type_ &&executor = {}) const noexcept {

        // If the cost of gap opening and extension is the same and we've mistakenly instantiated
        // the more memory-intensive `affine_gap_costs_t`, we can fall-back to the linearized version.
        if constexpr (is_same_type<gap_costs_t, affine_gap_costs_t>::value)
            if (gap_costs_.open == gap_costs_.extend) {
                linear_gap_costs_t linear_gap {gap_costs_.open};
                linearized_fallback_t linear_backend(substituter_, linear_gap, alloc_);
                return linear_backend(first, second, result_ref, executor);
            }

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, false>;
        similarity_memory_requirements_t requirements(                                 //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), SZ_MAX_REGISTER_WIDTH);

        // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
        if (requirements.bytes_per_cell <= 1 && requirements.max_diagonal_length < 16) {
            sz_u8_t result_u8 = std::numeric_limits<sz_u8_t>::max();
            status_t status = horizontal_u8_t {substituter_, gap_costs_, alloc_}(first, second, result_u8);
            if (status != status_t::success_k) return status;
            result_ref = result_u8;
        }

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        else if (requirements.bytes_per_cell <= 1) {
            sz_u8_t result_u8 = std::numeric_limits<sz_u8_t>::max();
            status_t status = diagonal_u8_t {substituter_, gap_costs_, alloc_}(first, second, result_u8, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_u8;
        }
        else if (requirements.bytes_per_cell == 2) {
            sz_u16_t result_u16 = std::numeric_limits<sz_u16_t>::max();
            status_t status = diagonal_u16_t {substituter_, gap_costs_, alloc_}(first, second, result_u16, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_u16;
        }
        else if (requirements.bytes_per_cell == 4) {
            sz_u32_t result_u32 = std::numeric_limits<sz_u32_t>::max();
            status_t status = diagonal_u32_t {substituter_, gap_costs_, alloc_}(first, second, result_u32, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_u32;
        }
        else if (requirements.bytes_per_cell == 8) {
            sz_u64_t result_u64 = std::numeric_limits<sz_u64_t>::max();
            status_t status = diagonal_u64_t {substituter_, gap_costs_, alloc_}(first, second, result_u64, executor);
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
    typename char_type_ = char,                    //
    typename gap_costs_type_ = linear_gap_costs_t, //
    typename allocator_type_ = dummy_alloc_t,      //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename enable_ = void                        //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct levenshtein_distance_utf8 {

    using char_t = char_type_;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using rune_allocator_t = typename allocator_traits_t::template rebind_alloc<sz_rune_t>;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = serialize_capability(capability_k);

    using horizontal_u8_t =                                                                           //
        horizontal_walker<sz_rune_t, sz_u8_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                          sz_minimize_distance_k, sz_similarity_global_k, capability_serialized_k>;
    using diagonal_u8_t =                                                                           //
        diagonal_walker<sz_rune_t, sz_u8_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_serialized_k>;
    using diagonal_u16_t =                                                                           //
        diagonal_walker<sz_rune_t, sz_u16_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u32_t =                                                                           //
        diagonal_walker<sz_rune_t, sz_u32_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u64_t =                                                                           //
        diagonal_walker<sz_rune_t, sz_u64_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;

    using linearized_fallback_t = levenshtein_distance<char_t, linear_gap_costs_t, allocator_t, capability_k>;
    using ascii_fallback_t = levenshtein_distance<char_t, gap_costs_t, allocator_t, capability_k>;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};
    mutable allocator_t alloc_ {};

    levenshtein_distance_utf8(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    levenshtein_distance_utf8(uniform_substitution_costs_t subs, gap_costs_t gaps,
                              allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, size_t &result_ref,
                        executor_type_ &&executor = {}) const noexcept {

        // If the cost of gap opening and extension is the same and we've mistakenly instantiated
        // the more memory-intensive `affine_gap_costs_t`, we can fall-back to the linearized version.
        if constexpr (is_same_type<gap_costs_t, affine_gap_costs_t>::value)
            if (gap_costs_.open == gap_costs_.extend) {
                linear_gap_costs_t linear_gap {gap_costs_.open};
                linearized_fallback_t linear_backend(substituter_, linear_gap, alloc_);
                return linear_backend(first, second, result_ref, executor);
            }

        // Check if the strings are entirely composed of ASCII characters,
        // and default to a simpler algorithm in that case.
        if (sz_isascii(first.data(), first.size()) && sz_isascii(second.data(), second.size()))
            return ascii_fallback_t {substituter_, gap_costs_, alloc_}(first, second, result_ref, executor);

        // Allocate some memory to expand UTF-8 strings into UTF-32.
        safe_vector<sz_rune_t, rune_allocator_t> unpacked_utf32(alloc_);
        if (unpacked_utf32.try_resize(first.size() + second.size()) != status_t::success_k)
            return status_t::bad_alloc_k;
        sz_rune_t *const first_data_utf32 = unpacked_utf32.data();
        sz_rune_t *const second_data_utf32 = first_data_utf32 + first.size();

        // Export into UTF-32 buffer.
        sz_rune_length_t rune_length;
        size_t first_length_utf32 = 0, second_length_utf32 = 0;
        for (size_t progress_utf8 = 0, progress_utf32 = 0; progress_utf8 < first.size();
             progress_utf8 += rune_length, ++progress_utf32, ++first_length_utf32) {
            sz_rune_parse(first.data() + progress_utf8, first_data_utf32 + progress_utf32, &rune_length);
            if (rune_length == sz_utf8_invalid_k) return status_t::invalid_utf8_k;
        }
        for (size_t progress_utf8 = 0, progress_utf32 = 0; progress_utf8 < second.size();
             progress_utf8 += rune_length, ++progress_utf32, ++second_length_utf32) {
            sz_rune_parse(second.data() + progress_utf8, second_data_utf32 + progress_utf32, &rune_length);
            if (rune_length == sz_utf8_invalid_k) return status_t::invalid_utf8_k;
        }

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, false>;
        similarity_memory_requirements_t requirements(                                 //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(sz_rune_t), SZ_MAX_REGISTER_WIDTH);

        span<sz_rune_t const> const first_utf32 {first_data_utf32, first_length_utf32};
        span<sz_rune_t const> const second_utf32 {second_data_utf32, second_length_utf32};

        // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
        if (requirements.bytes_per_cell <= 1 && requirements.max_diagonal_length < 16) {
            sz_u8_t result_u8 = std::numeric_limits<sz_u8_t>::max();
            status_t status = horizontal_u8_t {substituter_, gap_costs_, alloc_}(first_utf32, second_utf32, result_u8);
            if (status != status_t::success_k) return status;
            result_ref = result_u8;
        }

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        else if (requirements.bytes_per_cell <= 1) {
            sz_u8_t result_u8 = std::numeric_limits<sz_u8_t>::max();
            status_t status =                 diagonal_u8_t {substituter_, gap_costs_, alloc_}(first_utf32, second_utf32, result_u8,
executor);
            if (status != status_t::success_k) return status;
            result_ref = result_u8;
        }
        else if (requirements.bytes_per_cell == 2) {
            sz_u16_t result_u16 = std::numeric_limits<sz_u16_t>::max();
            status_t status =                 diagonal_u16_t {substituter_, gap_costs_, alloc_}(first_utf32, second_utf32, result_u16,
executor);
            if (status != status_t::success_k) return status;
            result_ref = result_u16;
        }
        else if (requirements.bytes_per_cell == 4) {
            sz_u32_t result_u32 = std::numeric_limits<sz_u32_t>::max();
            status_t status =                 diagonal_u32_t {substituter_, gap_costs_, alloc_}(first_utf32, second_utf32, result_u32,
executor);
            if (status != status_t::success_k) return status;
            result_ref = result_u32;
        }
        else if (requirements.bytes_per_cell == 8) {
            sz_u64_t result_u64 = std::numeric_limits<sz_u64_t>::max();
            status_t status =                 diagonal_u64_t {substituter_, gap_costs_, alloc_}(first_utf32, second_utf32, result_u64,
executor);
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
    typename char_type_ = char,                       //
    typename substituter_type_ = error_costs_32x32_t, //
    typename gap_costs_type_ = linear_gap_costs_t,    //
    typename allocator_type_ = dummy_alloc_t,         //
    sz_capability_t capability_ = sz_cap_serial_k,    //
    typename enable_ = void                           //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct needleman_wunsch_score {

    using char_t = char_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = serialize_capability(capability_k);

    using horizontal_i16_t =                                                         //
        horizontal_walker<char_t, sz_i16_t, substituter_t, gap_costs_t, allocator_t, //
                          sz_maximize_score_k, sz_similarity_global_k, capability_serialized_k>;
    using diagonal_i16_t =                                                         //
        diagonal_walker<char_t, sz_i16_t, substituter_t, gap_costs_t, allocator_t, //
                        sz_maximize_score_k, sz_similarity_global_k, capability_serialized_k>;
    using diagonal_i32_t =                                                         //
        diagonal_walker<char_t, sz_i32_t, substituter_t, gap_costs_t, allocator_t, //
                        sz_maximize_score_k, sz_similarity_global_k, capability_k>;
    using diagonal_i64_t =                                                         //
        diagonal_walker<char_t, sz_i64_t, substituter_t, gap_costs_t, allocator_t, //
                        sz_maximize_score_k, sz_similarity_global_k, capability_k>;

    substituter_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    needleman_wunsch_score(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    needleman_wunsch_score(substituter_t subs, gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, sz_ssize_t &result_ref,
                        executor_type_ &&executor = {}) const noexcept {

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, true>;
        similarity_memory_requirements_t requirements(                                 //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), SZ_MAX_REGISTER_WIDTH);

        // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
        status_t status = status_t::success_k;
        if (requirements.bytes_per_cell <= 2 && requirements.max_diagonal_length < 16) {
            sz_i16_t result_i16 = std::numeric_limits<sz_i16_t>::min();
            status = horizontal_i16_t {substituter_, gap_costs_, alloc_}(first, second, result_i16);
            if (status == status_t::success_k) result_ref = result_i16;
        }

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        else if (requirements.bytes_per_cell <= 2) {
            sz_i16_t result_i16 = std::numeric_limits<sz_i16_t>::min();
            status = diagonal_i16_t {substituter_, gap_costs_, alloc_}(first, second, result_i16, executor);
            if (status == status_t::success_k) result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            sz_i32_t result_i32 = std::numeric_limits<sz_i32_t>::min();
            status = diagonal_i32_t {substituter_, gap_costs_, alloc_}(first, second, result_i32, executor);
            if (status == status_t::success_k) result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            sz_i64_t result_i64 = std::numeric_limits<sz_i64_t>::min();
            status = diagonal_i64_t {substituter_, gap_costs_, alloc_}(first, second, result_i64, executor);
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
    typename char_type_ = char,                       //
    typename substituter_type_ = error_costs_32x32_t, //
    typename gap_costs_type_ = linear_gap_costs_t,    //
    typename allocator_type_ = dummy_alloc_t,         //
    sz_capability_t capability_ = sz_cap_serial_k,    //
    typename enable_ = void                           //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct smith_waterman_score {

    using char_t = char_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = serialize_capability(capability_k);

    using horizontal_i16_t =                                                         //
        horizontal_walker<char_t, sz_i16_t, substituter_t, gap_costs_t, allocator_t, //
                          sz_maximize_score_k, sz_similarity_local_k, capability_serialized_k>;
    using diagonal_i16_t =                                                         //
        diagonal_walker<char_t, sz_i16_t, substituter_t, gap_costs_t, allocator_t, //
                        sz_maximize_score_k, sz_similarity_local_k, capability_serialized_k>;
    using diagonal_i32_t =                                                         //
        diagonal_walker<char_t, sz_i32_t, substituter_t, gap_costs_t, allocator_t, //
                        sz_maximize_score_k, sz_similarity_local_k, capability_k>;
    using diagonal_i64_t =                                                         //
        diagonal_walker<char_t, sz_i64_t, substituter_t, gap_costs_t, allocator_t, //
                        sz_maximize_score_k, sz_similarity_local_k, capability_k>;

    substituter_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    smith_waterman_score(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    smith_waterman_score(substituter_t subs, gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, sz_ssize_t &result_ref,
                        executor_type_ &&executor = {}) const noexcept {

        size_t const first_length = first.size();
        size_t const second_length = second.size();
        if (first_length == 0 || second_length == 0) {
            result_ref = 0;
            return status_t::success_k;
        }

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, true>;
        similarity_memory_requirements_t requirements(                                 //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), SZ_MAX_REGISTER_WIDTH);

        // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
        if (requirements.bytes_per_cell <= 2 && requirements.max_diagonal_length < 16) {
            sz_i16_t result_i16 = std::numeric_limits<sz_i16_t>::min();
            status_t status = horizontal_i16_t {substituter_, gap_costs_, alloc_}(first, second, result_i16);
            if (status != status_t::success_k) return status;
            result_ref = result_i16;
        }

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        else if (requirements.bytes_per_cell <= 2) {
            sz_i16_t result_i16 = std::numeric_limits<sz_i16_t>::min();
            status_t status = diagonal_i16_t {substituter_, gap_costs_, alloc_}(first, second, result_i16, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            sz_i32_t result_i32 = std::numeric_limits<sz_i32_t>::min();
            status_t status = diagonal_i32_t {substituter_, gap_costs_, alloc_}(first, second, result_i32, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            sz_i64_t result_i64 = std::numeric_limits<sz_i64_t>::min();
            status_t status = diagonal_i64_t {substituter_, gap_costs_, alloc_}(first, second, result_i64, executor);
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
 *         differentiating multi-threaded and single-threaded cases.
 *         For pairs of very large strings, all cores cooperate to compute one distance maximizing
 *         cache hits. For smaller strings, each core computes its own distance.
 */
template <                                     //
    typename score_type_,                      //
    typename scoring_type_,                    //
    typename first_strings_type_,              //
    typename second_strings_type_,             //
    typename results_type_,                    //
    typename executor_type_ = dummy_executor_t //
    >
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && executor_like<executor_type_> && indexed_results_like<results_type_>
#endif
status_t _score_in_parallel(                                                                                       //
    scoring_type_ &&scoring, first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
    results_type_ &&results, size_t substitute_magnitude, size_t gap_magnitude,                                    //
    executor_type_ &&executor = {}, cpu_specs_t specs = {}) noexcept {

    using score_t = score_type_;
    constexpr bool score_is_signed_k = std::is_signed_v<score_t>;
    using similarity_memory_requirements_t = similarity_memory_requirements<size_t, score_is_signed_k>;
    using char_t = typename scoring_type_::char_t;
    using gap_costs_t = typename scoring_type_::gap_costs_t;

    auto first_size = first_strings.size();
    auto second_size = second_strings.size();
    sz_assert_(first_size == second_size && "Expect equal number of strings");

    // Use an atomic to store any error encountered.
    std::atomic<status_t> error {status_t::success_k};

    // ? There may be a huge variance in the lengths of the strings,
    // ? so we need to use a dynamic schedule.
    executor.for_n_dynamic(first_size, [&](size_t i) noexcept {
        if (error.load() != status_t::success_k) return;
        score_t result = 0;
        auto const &first = first_strings[i];
        auto const &second = second_strings[i];

        // ! Longer strings will be handled separately
        similarity_memory_requirements_t requirements(                    //
            first.size(), second.size(),                                  //
            gap_type<gap_costs_t>(), substitute_magnitude, gap_magnitude, //
            sizeof(char_t), SZ_MAX_REGISTER_WIDTH);

        if (requirements.total >= specs.l1_bytes) return;
        status_t status = scoring({first.data(), first.size()}, {second.data(), second.size()}, result);
        if (status == status_t::success_k) { results[i] = result; }
        else { error.store(status); }
    });

    // Now handle the longer strings.
    for (size_t i = 0; i < first_size && error.load() == status_t::success_k; ++i) {
        score_t result = 0;
        auto const &first = first_strings[i];
        auto const &second = second_strings[i];
        similarity_memory_requirements_t requirements(                    //
            first.size(), second.size(),                                  //
            gap_type<gap_costs_t>(), substitute_magnitude, gap_magnitude, //
            sizeof(char_t), SZ_MAX_REGISTER_WIDTH);

        if (requirements.total < specs.l1_bytes) continue;
        status_t status = scoring({first.data(), first.size()}, {second.data(), second.size()}, result, executor);
        if (status == status_t::success_k) { results[i] = result; }
        else { error.store(status); }
    }
    return error.load();
}

template <                         //
    typename score_type_,          //
    typename scoring_type_,        //
    typename first_strings_type_,  //
    typename second_strings_type_, //
    typename results_type_         //
    >
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && indexed_results_like<results_type_>
#endif
status_t _score_sequentially(                                                                                      //
    scoring_type_ &&scoring, first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
    results_type_ &&results) noexcept {

    using score_t = score_type_;

    auto first_size = first_strings.size();
    auto second_size = second_strings.size();
    sz_assert_(first_size == second_size && "Expect equal number of strings");

    for (size_t i = 0; i < first_size; ++i) {
        score_t result = 0;
        auto const &first = first_strings[i];
        auto const &second = second_strings[i];
        status_t status = scoring({first.data(), first.size()}, {second.data(), second.size()}, result);
        if (status == status_t::success_k) { results[i] = result; }
        else { return status; }
    }
    return status_t::success_k;
}

template <                       //
    typename char_type_,         //
    typename gap_costs_type_,    //
    typename allocator_type_,    //
    sz_capability_t capability_, //
    typename enable_             //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct levenshtein_distances {

    using char_t = char_type_;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    using scoring_t = levenshtein_distance<char_t, gap_costs_t, allocator_t, capability_k>;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    levenshtein_distances(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    levenshtein_distances(uniform_substitution_costs_t subs, gap_costs_t gaps,
                          allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results) const noexcept {
        return _score_sequentially<size_t>(               //
            scoring_t {substituter_, gap_costs_, alloc_}, //
            first_strings, second_strings, std::forward<results_type_>(results));
    }

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_,
              typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_> && indexed_results_like<results_type_>
#endif
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, executor_type_ &&executor,
                        cpu_specs_t const &specs = {}) const noexcept {
        return _score_in_parallel<size_t>(                                       //
            scoring_t {substituter_, gap_costs_, alloc_},                        //
            first_strings, second_strings, std::forward<results_type_>(results), //
            substituter_.magnitude(), gap_costs_.magnitude(), executor, specs);
    }
};

template <                       //
    typename char_type_,         //
    typename gap_costs_type_,    //
    typename allocator_type_,    //
    sz_capability_t capability_, //
    typename enable_             //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct levenshtein_distances_utf8 {

    using char_t = char_type_;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    using scoring_t = levenshtein_distance_utf8<char_t, gap_costs_t, allocator_t, capability_k>;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    levenshtein_distances_utf8(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    levenshtein_distances_utf8(uniform_substitution_costs_t subs, gap_costs_t gaps,
                               allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results) const noexcept {
        return _score_sequentially<size_t>(               //
            scoring_t {substituter_, gap_costs_, alloc_}, //
            first_strings, second_strings, std::forward<results_type_>(results));
    }

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_,
              typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_> && indexed_results_like<results_type_>
#endif
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, executor_type_ &&executor,
                        cpu_specs_t const &specs = {}) const noexcept {
        return _score_in_parallel<size_t>(                                       //
            scoring_t {substituter_, gap_costs_, alloc_},                        //
            first_strings, second_strings, std::forward<results_type_>(results), //
            substituter_.magnitude(), gap_costs_.magnitude(), executor, specs);
    }
};

template <                       //
    typename char_type_,         //
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

    using char_t = char_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    using scoring_t = needleman_wunsch_score<char_t, substituter_t, gap_costs_t, allocator_t, capability_k>;

    substituter_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    needleman_wunsch_scores(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    needleman_wunsch_scores(substituter_t subs, gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results) const noexcept {
        return _score_sequentially<ssize_t>(              //
            scoring_t {substituter_, gap_costs_, alloc_}, //
            first_strings, second_strings, std::forward<results_type_>(results));
    }

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_,
              typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_> && indexed_results_like<results_type_>
#endif
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, executor_type_ &&executor,
                        cpu_specs_t const &specs = {}) const noexcept {
        return _score_in_parallel<ssize_t>(                                      //
            scoring_t {substituter_, gap_costs_, alloc_},                        //
            first_strings, second_strings, std::forward<results_type_>(results), //
            substituter_.magnitude(), gap_costs_.magnitude(), executor, specs);
    }
};

template <                       //
    typename char_type_,         //
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

    using char_t = char_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    using scoring_t = smith_waterman_score<char_t, substituter_t, gap_costs_t, allocator_t, capability_k>;

    substituter_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    smith_waterman_scores(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    smith_waterman_scores(substituter_t subs, gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results) const noexcept {
        return _score_sequentially<ssize_t>(              //
            scoring_t {substituter_, gap_costs_, alloc_}, //
            first_strings, second_strings, std::forward<results_type_>(results));
    }

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_,
              typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_> && indexed_results_like<results_type_>
#endif
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, executor_type_ &&executor,
                        cpu_specs_t const &specs = {}) const noexcept {
        return _score_in_parallel<ssize_t>(                                      //
            scoring_t {substituter_, gap_costs_, alloc_},                        //
            first_strings, second_strings, std::forward<results_type_>(results), //
            substituter_.magnitude(), gap_costs_.magnitude(), executor, specs);
    }
};

#pragma endregion

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_SERIAL_HPP_
