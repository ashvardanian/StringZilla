/**
 *  @brief  Hardware-accelerated feature extractions for string collections.
 *  @file   features.hpp
 *  @author Ash Vardanian
 *
 *  The `sklearn.feature_extraction` module for @b TF-IDF, `CountVectorizer`, and `HashingVectorizer`
 *  is one of the most commonly used in the industry due to its extreme flexibility. It can:
 *
 *  - Tokenize by words, N-grams, or in-word N-grams.
 *  - Use arbitrary Regular Expressions as word separators.
 *  - Return matrices of different types, normalized or not.
 *  - Exclude "stop words" and remove ASCII and Unicode accents.
 *  - Dynamically build a vocabulary or use a fixed list/dictionary.
 *
 *  That level of flexibility is not feasible for a hardware-accelerated SIMD library, but we
 *  can provide a set of APIs that can be used to build such a library on top of StringCuZilla.
 *  That functionality can reuse our @b Trie data-structure for vocabulary building histograms.
 *
 *  In this file, we mostly focus on batch-level hashing operations, similar to the `intersect.h`
 *  module. There, we cross-reference two sets of strings, and here we only analyze one at a time.
 *
 *  - The text comes in pre-tokenized form, as a stream, not even indexed-lookup is needed,
 *    unlike the `sz_sequence_t` in `sz_intersect` APIs.
 *  - We scatter those tokens into the output in multiple forms:
 *
 *    - output hashes into a continuous buffer.
 *    - output hashes into a hash-map with counts.
 *    - output hashes into a high-dimensional bit-vector.
 *
 */
#ifndef STRINGZILLA_FEATURES_HPP_
#define STRINGZILLA_FEATURES_HPP_

#include "stringzilla/memory.h"  // `sz_move`
#include "stringzilla/types.hpp" // `sz::error_cost_t`

#include <limits>   // `std::numeric_limits` for numeric types
#include <iterator> // `std::iterator_traits` for iterators
