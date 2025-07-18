/**
 *  @brief  Hardware-accelerated UTF-8 segments, locating graphemes, word- and sentence-boundaries.
 *  @file   segment.hpp
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_runes_count(bytes_view, runes_limit)` → (bytes_skipped, runes_found)
 *  - `sz_runes_parse(bytes_view, runes_span)` → (bytes_skipped, runes_found)
 *  - `sz_clusters_count(runes_view, grapheme_clusters_limit)` → (bytes_skipped, runes_found)
 *  - `sz_clusters_parse(runes_view, grapheme_clusters_span)` → (bytes_skipped, runes_found)
 *
 *  The first: counts the number of UTF-8 runes in a given string, up to a given limit.
 *  The second: parses a UTF-8 string into an array of UTF-32 runes - optimized for batch-decoding of 64 runes/call.
 *  To render the text, however, we need to know the size of different grapheme clusters, which is defined by:
 *
 *  - UAX #29 "Unicode Text Segmentation" rules: https://unicode.org/reports/tr29/
 *  - UTS #51 "Unicode Emoji" rules: https://www.unicode.org/reports/tr51/
 *
 *  @see "Blazing fast Unicode-aware ILIKE with AVX-512" in Sneller:
 *       https://sneller.ai/blog/accelerating-ilike-using-avx-512
 *  @see For fast any-to-any transcoding: https://github.com/simdutf/simdutf
 *  @see For UTF-8 validation: https://github.com/lemire/fastvalidate-utf-8
 *
 */
#ifndef STRINGZILLA_SEGMENTS_HPP_
#define STRINGZILLA_SEGMENTS_HPP_

#include "types.h"

#include "compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

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
    sz_unused_(can_be_ascii + can_be_utf8 + can_be_utf16 + can_be_utf32);
    sz_unused_(text && length);
    return sz_false_k;
}
