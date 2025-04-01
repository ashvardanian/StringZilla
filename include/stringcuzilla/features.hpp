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
 *  can provide a set of APIs that can be used to build such a library on top of StringZilla.
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

#include "types.h"

#include "compare.h" // `sz_compare`
#include "memory.h"  // `sz_copy`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief  Faster @b arg-sort for an arbitrary @b string sequence, using QuickSort.
 *          Outputs the @p order of elements in the immutable @p sequence, that would sort it.
 *
 *  @param[in] sequence Immutable sequence of strings to sort.
 *  @param[in] alloc Optional memory allocator for temporary storage.
 *  @param[out] order Output permutation that sorts the elements.
 *
 *  @retval `sz_success_k` if the operation was successful.
 *  @retval `sz_bad_alloc_k` if the operation failed due to memory allocation failure.
 *  @pre The @p order array must fit at least `sequence->count` integers.
 *  @post The @p order array will contain a valid permutation of `[0, sequence->count - 1]`.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/features.h>
 *      int main() {
 *          char const *strings[] = {"banana", "apple", "cherry"};
 *          sz_sequence_t sequence;
 *          sz_sequence_from_null_terminated_strings(strings, 3, &sequence);
 *          sz_sorted_idx_t order[3];
 *          sz_status_t status = sz_sequence_argsort(&sequence, NULL, order);
 *          return status == sz_success_k && order[0] == 1 && order[1] == 0 && order[2] == 2 ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note   The algorithm has linear memory complexity, quadratic worst-case and log-linear average time complexity.
 *  @see    https://en.wikipedia.org/wiki/Quicksort
 *
 *  @note   This algorithm is @b unstable: equal elements may change relative order.
 *  @sa     sz_sequence_argsort_stabilize
 *
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_sequence_argsort_serial, sz_sequence_argsort_skylake, sz_sequence_argsort_sve
 */
SZ_DYNAMIC sz_status_t sz_sequence_argsort(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                           sz_sorted_idx_t *order);

enum sz_encoding_t {
    sz_encoding_unknown_k = 0,
    sz_encoding_ascii_k = 1,
    sz_encoding_utf8_k = 2,
    sz_encoding_utf16_k = 3,
    sz_encoding_utf32_k = 4,
    sz_encoding_jwt_k = 5,
    sz_encoding_base64_k = 6,
    // Low priority encodings:
    sz_encoding_utf8bom_k = 7,
    sz_encoding_utf16le_k = 8,
    sz_encoding_utf16be_k = 9,
    sz_encoding_utf32le_k = 10,
    sz_encoding_utf32be_k = 11,
};

// Character Set Detection is one of the most commonly performed operations in data processing with
// [Chardet](https://github.com/chardet/chardet), [Charset Normalizer](https://github.com/jawah/charset_normalizer),
// [cChardet](https://github.com/PyYoshi/cChardet) being the most commonly used options in the Python ecosystem.
// All of them are notoriously slow.
//
// Moreover, as of October 2024, UTF-8 is the dominant character encoding on the web, used by 98.4% of websites.
// Other have minimal usage, according to [W3Techs](https://w3techs.com/technologies/overview/character_encoding):
// - ISO-8859-1: 1.2%
// - Windows-1252: 0.3%
// - Windows-1251: 0.2%
// - EUC-JP: 0.1%
// - Shift JIS: 0.1%
// - EUC-KR: 0.1%
// - GB2312: 0.1%
// - Windows-1250: 0.1%
// Within programming language implementations and database management systems, 16-bit and 32-bit fixed-width encodings
// are also very popular and we need a way to efficiently differentiate between the most common UTF flavors, ASCII, and
// the rest.
//
// One good solution is the [simdutf](https://github.com/simdutf/simdutf) library, but it depends on the C++ runtime
// and focuses more on incremental validation & transcoding, rather than detection.
//
// So we need a very fast and efficient way of determining
SZ_PUBLIC sz_bool_t sz_detect_encoding(sz_cptr_t text, sz_size_t length) {
    // https://github.com/simdutf/simdutf/blob/master/src/icelake/icelake_utf8_validation.inl.cpp
    // https://github.com/simdutf/simdutf/blob/603070affe68101e9e08ea2de19ea5f3f154cf5d/src/icelake/icelake_from_utf8.inl.cpp#L81
    // https://github.com/simdutf/simdutf/blob/603070affe68101e9e08ea2de19ea5f3f154cf5d/src/icelake/icelake_utf8_common.inl.cpp#L661
    // https://github.com/simdutf/simdutf/blob/603070affe68101e9e08ea2de19ea5f3f154cf5d/src/icelake/icelake_utf8_common.inl.cpp#L788

    // We can implement this operation simpler & differently, assuming most of the time continuous chunks of memory
    // have identical encoding. With Russian and many European languages, we generally deal with 2-byte codepoints
    // with occasional 1-byte punctuation marks. In the case of Chinese, Japanese, and Korean, we deal with 3-byte
    // codepoints. In the case of emojis, we deal with 4-byte codepoints.
    // We can also use the idea, that misaligned reads are quite cheap on modern CPUs.
    int can_be_ascii = 1, can_be_utf8 = 1, can_be_utf16 = 1, can_be_utf32 = 1;
    sz_unused(can_be_ascii + can_be_utf8 + can_be_utf16 + can_be_utf32);
    sz_unused(text && length);
    return sz_false_k;
}

#pragma endregion // Core API

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_FEATURES_HPP_
