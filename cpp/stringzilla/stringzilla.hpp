/**
 *  @file cpp/stringzilla/stringzilla.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief StringZilla SDK for C++20 and newer, improving over the performance of
 *      @c std::string_view and @c std::string, mostly for substring search.
 *
 *  Every header stands alone, and this one includes them all. The containers, which call no kernel
 *  and need no library, live in <tt>include/stringzilla/</tt>:
 *
 *  - <tt>types.hpp</tt> - @c status_t, @c expected, the allocators and @c vector.
 *  - <tt>sets.hpp</tt> - @c byteset_t, @c look_up_table_t and the character sets.
 *  - <tt>string.hpp</tt> - @c string_view_t, @c string_span_t and the owning @c string.
 *  - <tt>strings.hpp</tt> - @c strings_view_t and the owned @c strings column.
 *
 *  The verbs, which link the library and mirror the C families, live in <tt>cpp/stringzilla/</tt>:
 *
 *  - <tt>devices.hpp</tt> - @c device_t, @c default_capabilities and @c make_unified_alloc.
 *  - <tt>compare.hpp</tt> - @c starts_with, @c ends_with, @c remove_prefix and @c remove_suffix.
 *  - <tt>find.hpp</tt> - @c find, @c rfind, the byte-set searches, and the ranges of matches and splits.
 *  - <tt>hash.hpp</tt> - @c hash, @c bytesum, @c fill_random, @c hash_t and @c std::hash.
 *  - <tt>strings_verbs.hpp</tt> - @c gather, @c join, @c lookup, @c split_into and the other column verbs.
 *  - <tt>utf8_runes.hpp</tt>, <tt>utf8_tokens.hpp</tt>, <tt>utf8_wordbreaks.hpp</tt>, <tt>utf8_graphemes.hpp</tt>,
 *    <tt>utf8_sentences.hpp</tt>, <tt>utf8_linebreaks.hpp</tt>, <tt>utf8_norm.hpp</tt>, <tt>utf8_uncased.hpp</tt> -
 *    the UTF-8 counts, ranges, normalization and case-insensitive verbs.
 *  - <tt>sort.hpp</tt> - @c argsort and @c argsort_utf8_uncased.
 *  - <tt>intersect.hpp</tt> - @c intersect.
 *  - <tt>levenshtein.hpp</tt> - @c levenshtein_engine_t, edit distances of query batches.
 *  - <tt>ngrams.hpp</tt> - @c ngrams_engine_t, N-gram overlaps and MinHash sketches.
 *  - <tt>substrings.hpp</tt> - @c substrings_engine_t, multi-pattern substring search.
 *  - <tt>subdocuments.hpp</tt> - @c subdocuments_engine_t, JSON and XML path queries.
 *  - <tt>regex.hpp</tt> - @c regex_engine_t, multi-pattern regular expressions.
 *
 *  @see Compile Health: https://artificial-mind.net/projects/compile-health/
 *  @see StringZilla: https://github.com/ashvardanian/StringZilla/blob/main/README.md
 *  @see C++ Standard String: https://en.cppreference.com/w/cpp/header/string
 */
#ifndef STRINGZILLA_HPP_
#define STRINGZILLA_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/types.hpp"
#include "stringzilla/sets.hpp"
#include "stringzilla/string.hpp"
#include "stringzilla/strings.hpp"

#include "stringzilla/devices.hpp"
#include "stringzilla/compare.hpp"
#include "stringzilla/find.hpp"
#include "stringzilla/hash.hpp"
#include "stringzilla/strings_verbs.hpp"
#include "stringzilla/utf8_runes.hpp"
#include "stringzilla/utf8_tokens.hpp"
#include "stringzilla/utf8_wordbreaks.hpp"
#include "stringzilla/utf8_graphemes.hpp"
#include "stringzilla/utf8_sentences.hpp"
#include "stringzilla/utf8_linebreaks.hpp"
#include "stringzilla/utf8_norm.hpp"
#include "stringzilla/utf8_uncased.hpp"
#include "stringzilla/sort.hpp"
#include "stringzilla/intersect.hpp"
#include "stringzilla/levenshtein.hpp"
#include "stringzilla/ngrams.hpp"
#include "stringzilla/substrings.hpp"
#include "stringzilla/subdocuments.hpp"
#include "stringzilla/regex.hpp"

#endif // STRINGZILLA_HPP_
